from lib.proxy.regression import DropoutRegressor, EnsembleRegressor

def get_proxy_model(args, tokenizer):
    if args.proxy_uncertainty == "dropout":
        proxy = DropoutRegressor(args, tokenizer)
    elif args.proxy_uncertainty == "ensemble":
        proxy = EnsembleRegressor(args, tokenizer)
    return proxy