from lib.generator.gfn import FMGFlowNetGenerator, TBGFlowNetGenerator


def get_generator(args, tokenizer):
    if not args.gen_do_explicit_Z:
        return FMGFlowNetGenerator(args, tokenizer)
    else:
        return TBGFlowNetGenerator(args, tokenizer)