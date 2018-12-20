def create_model(opt):
    model = None
    if opt.model == 'GAN':
        assert(opt.dataset_mode == 'unaligned')
        from .SRN_GAN import SRN_GANModel
        model = SRN_GANModel()

    else:
        raise NotImplementedError('model [%s] not implemented' % opt.model)
    model.initialize(opt)
    print("model [%s] war created" % (model.name()))
    return model
