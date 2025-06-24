import torch
import torchvision
from torchvision import transforms
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import torch.nn.functional as F
import torch.distributed as dist
import cv2
import os
import csv
import math
import argparse
import pickle
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from safetensors.torch import load_file
from kiui.op import safe_normalize

import sys;sys.path.append(".")
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.t5 import T5Embedder
from autoregressive.models.nv_campos_iwc_gpt import NVCGPT_models
from autoregressive.models.generate_nv_iwc import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

gso_prompts = [
    "Round vintage metal alarm clock with loop handle, supporting legs, yellowed face, black numerals, and hands.", 
    "Children's backpack with a light blue background, featuring colorful vehicle prints, dual compartments, mesh side pockets, padded straps, and a black back panel.", 
    "Handbell with a polished wooden handle and a flared metallic bell, featuring a smooth finish.", 
    "Wooden toy building blocks in various shapes and colors, including beige, red, blue, and green.", 
    "Stackable wooden toy with three colorful, rounded layers in orange, red, and yellow, topped with a white egg.", 
    "White cubic product packaging box with green label, black text, and branding for Perricone MD skincare.", 
    "Gray plush elephant toy with large floppy ears, white accents, a burgundy ribbon, and a tufted tail.", 
    "Wooden figurine with a round head, blue shirt, black vest, blue pants, and glasses.", 
    "Wooden figurine with white hair, glasses, a burgundy top, light pink pants, and beige shoes.", 
    "Light brown fedora hat with a textured surface, creased crown, and a white ribbon band.", 
    "Dark brown leather loafer with a slip-on design, stitched detailing, cushioned interior, and rubber sole.", 
    "Realistic lion figurine with a detailed brown mane, tan body, muscular limbs, and a long tail.", 
    "Camouflage-patterned tactical bag with a dual-compartment design, zippered closure, side loops, and a carrying handle.", 
    "Red-hatted figurine in blue overalls, red shirt, white gloves, brown shoes, and mustache.", 
    "Turquoise squeeze tube labeled 'Oil of Morocco Argan Oil' with yellow accents and black text.", 
    "Yellow wooden school bus toy with black stripes, white wheels, and red stop sign detailing.", 
    "Yellow toy truck with black wheels, a smiling face, a driver figure, and colorful blocks stacked on top.", 
    "Black boat shoe with white soles, contrasting stitching, and lace-up design.", 
    "Beige high-top boots with tan accents, gold eyelets, and a cushioned ankle collar for added comfort.", 
    "Gray ankle boots with a smooth finish, side zipper, and padded ankle area for extra comfort.", 
    "Brown sneakers with textured fabric, lace-up closure, and the Adidas logo on the back and tongue.", 
    "White ceramic dish placed in a wooden holder with angled sides, designed as a shallow rectangular container.", 
    "Wooden sofa with a purple cushion, featuring simple, curved armrests and a striped wooden frame.", 
    "Blue metal file organizer with five vertical slots for holding documents or files in place.", 
    "Wooden toy set featuring colorful stacking blocks in shapes like squares, circles, and triangles.", 
    "Stack of six colorful plastic cups, with red, orange, yellow, green, and blue layers, cylindrical shape.", 
    "A white porcelain teapot with a round body, a spout, a handle, and a round lid.", 
    "A silver, rectangular two-slice toaster with a transparent top and side button controls.", 
    "a colorful toy train made of blocks, featuring red, blue, yellow, green, and orange components with a rope for pulling.", 
    "A colorful plastic toy turtle with gears on its back, yellow wheels.", 
]


def get_rays(pose, h, w, fovy, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d


def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = NVCGPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        num_viewes=args.num_views,
    ).to(device=device, dtype=precision)
    gpt_model.use_checkpoint = False

    if ".safetensor" in args.gpt_ckpt:
        checkpoint = {"model": load_file(args.gpt_ckpt)}

        new_token_embeddings = torch.ones_like(gpt_model.tok_embeddings.weight.data) * 1e-6
        new_token_embeddings[:args.codebook_size, :] = checkpoint["model"]["tok_embeddings.weight"].data[:args.codebook_size, :]
        gpt_model.tok_embeddings.weight.data.copy_(new_token_embeddings)
        del checkpoint["model"]["tok_embeddings.weight"]

        new_uncond_embedding = checkpoint["model"]["condition_mlp.uncond_embedding"].data[:1024, :]
        gpt_model.condition_mlp.uncond_embedding.data.copy_(new_uncond_embedding)
        del checkpoint["model"]["condition_mlp.uncond_embedding"]
    else:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    # TODO: after deepspeed train need this.
    original_model_weight = OrderedDict()
    for k, v in model_weight.items():
        original_model_weight[k[7:]] = v

    gpt_model.load_state_dict(original_model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print(f"t5 model is loaded")

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-gso-size-{args.image_size}-size-{args.image_size}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    if args.kvcache:
        folder_name = f"{folder_name}-kvcache"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(f"{sample_folder_dir}/images", exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}/images")
    dist.barrier()
    
    # list gso
    gso_paths = os.listdir("dataset/gso")
    gso_paths = sorted(gso_paths)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    num_fid_samples = min(args.num_fid_samples, len(gso_paths))
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")

    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.ToTensor(),
        # transforms.Resize((args.image_size, args.image_size)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    sample_index = [0, 4, 8, 12] # 4 views
    # sample_index = [0, 2, 4, 6, 8, 10, 12, 14] # 8 views
    num_viewes = len(sample_index)

    # Select top-num-views' camera poses
    c_camera = torch.from_numpy(np.load(f"camera_pose.npy")).unsqueeze(0)  # B, num_views, 4, 4
    c_camera = c_camera[:, sample_index, ...].squeeze(0)
    rays_plucker = []
    for c in c_camera:
        rays_o, rays_d = get_rays(c.float(), args.image_size, args.image_size, 49.1)
        rays_plucker.append(torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1))
    rays_plucker = torch.stack(rays_plucker)
    rays_plucker = rays_plucker.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    rays_plucker = rays_plucker.to(torch.bfloat16)

    # Setup camera embedding
    with torch.no_grad():
        camera_embeddings = gpt_model.cam_patchify(rays_plucker) * gpt_model.camera_gamma
    c_camera = camera_embeddings.flatten(0, 1).unsqueeze(0)

    for gso_name in tqdm(gso_paths):
        print(gso_name)
        if ".txt" in gso_name:
            continue

        gso_path = os.path.join("dataset/gso", gso_name)
        reference_image = cv2.imread(os.path.join(gso_path, "render_persp", "front.png"))
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        reference_image = torch.from_numpy(reference_image).unsqueeze(0).to(device, non_blocking=True)

        # get reference image indices
        with torch.no_grad():
            reference_image = transform(reference_image.permute(0, 3, 1, 2).contiguous() / 255)
            _, _, [_, _, reference_indices] = vq_model.encode(reference_image)
        reference_indices = reference_indices.unsqueeze(0)

        # get reference image condition
        reference_image_cond = gpt_model.tok_embeddings(reference_indices)
        image_condition_embed = [reference_image_cond]
        for i in range(num_viewes - 1):
            warpped_image = gpt_model.image_warp(reference_image_cond, camera_embeddings[i+1].unsqueeze(0))
            image_condition_embed.append(warpped_image)
        image_condition_embed = torch.cat(image_condition_embed, dim=1)

        # Sample inputs:
        prompt_batch = ["generate multi view images of the following image."]
        caption_embs, emb_masks = t5_model.get_text_embeddings(prompt_batch)
        c_camera = c_camera.type_as(caption_embs)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                # prompt_cur = prompt_batch[idx]
                # print(f'  prompt {idx} token len: {valid_num} : {prompt_cur}')
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices) * args.num_views, args.codebook_embed_dim, latent_size, latent_size]
        ref_qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]

        # [0]: warped image condition, [1]: text embedding
        c_indices = [image_condition_embed, c_indices]

        if args.kvcache:
            index_sample = generate(
                gpt_model, c_indices, latent_size ** 2 * args.num_views, 
                cond_camera=c_camera, 
                emb_masks=c_emb_masks, 
                cfg_scale=args.cfg_scale, 
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
            )
        else:
            raise ValueError

        if gpt_model.model_type == 't&cam2i':
            index_sample = index_sample.reshape(-1, latent_size ** 2)

        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        # samples = samples.permute(0, 2, 3, 1)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).cpu() / 255

        reference_samples = vq_model.decode_code(reference_indices, ref_qzshape) # output value is between [-1, 1]
        reference_samples = torch.clamp(127.5 * reference_samples + 128.0, 0, 255).cpu() / 255

        # Save samples to disk as individual .png files
        b, c, h, w = reference_samples.shape
        samples = samples.reshape(len(prompt_batch), -1, c, h, w)
        sample = samples[0]
        torchvision.utils.save_image(sample, f"{sample_folder_dir}/images/{gso_name}_tcam2i_has_ref.png")
        torchvision.utils.save_image(reference_samples, f"{sample_folder_dir}/images/{gso_name}_ref.png")

        gpt_model.reset_caches(dtype=precision)

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvcache", action='store_true', default=False)
    parser.add_argument("--t5-path", type=str, default='/mnt/nasv3/hjk/huggingface/flan-t5-xl')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--camera-pose-path", type=str, default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(NVCGPT_models.keys()), default="NVCGPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 't&cam2i'], default="t&cam2i", help="text / image -> multiview-image")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="pretrained_models/vq_ds16_tcam2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--sample-dir", type=str, default="samples_objaverse_nv_ray", help="samples_objaverse")
    parser.add_argument("--per-proc-batch-size", type=int, default=1)
    parser.add_argument("--num-fid-samples", type=int, default=30000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--drop-text", type=bool, default=False, help="whter to drop text condition")
    parser.add_argument("--drop-shape", type=bool, default=False, help="whter to drop shape condition")
    # parser.
    args = parser.parse_args()
    main(args)
