from dataset.mvdataset import build_t_cam2i_code, build_t_ray2i_nv


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 't&cam2i':
        return build_t_cam2i_code(args, **kwargs)
    if args.dataset == 't&ray2i_nv':
        return build_t_ray2i_nv(args, **kwargs)

    raise ValueError(f'dataset {args.dataset} is not supported')
