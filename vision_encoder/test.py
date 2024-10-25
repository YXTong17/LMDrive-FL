# from lavis.common.config import Config
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")

#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#                 "--options",
#                 nargs="+",
#                 help="override some settings in the used config, the key-value pair "
#                 "in xxx=yyy format will be merged into config file (deprecate), "
#                 "change to --cfg-options instead.",
#             )
#     args = parser.parse_args()
#             # if 'LOCAL_RANK' not in os.environ:
#             #     os.environ['LOCAL_RANK'] = str(args.local_rank)
#     print(args)
#     return args
# cfg = Config(parse_args())
# print(cfg)
a = 4
for  i in zip(a):
    print(i)