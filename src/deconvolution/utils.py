import src.deconvolution as deconv
def run_deconvolution(method,**kwargs):
    
    case = {
        'bmind': deconv.bmind.run_bMIND,
        'gan': deconv.gan.run_DLunmix_GAN,
        'gp': deconv.gp.run_DLunmix_GP,
        'gp_nb': deconv.gp_nb.run_DLunmix_NBGP
    }

    return case[method](**kwargs)