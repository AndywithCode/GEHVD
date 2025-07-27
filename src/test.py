from transformers import AutoModelForSeq2SeqLM
import json

# print(data)

# __encoder = AutoModelForSeq2SeqLM.from_pretrained('codet5p-770m', trust_remote_code=True).encoder
# print(__encoder)

# 加载checkpoint模型
checkpoint_path = "path/to/saved_checkpoint.ckpt"
model = MyModel.load_from_checkpoint(checkpoint_path)
# 可选：调整测试参数
model.pred_step = 1000
model.eval()
# 初始化Trainer（注意关闭训练相关配置）
test_trainer = pl.Trainer(
    gpus=gpu,
    logger=[tensorlogger],  # 保留日志记录
    callbacks=[print_epoch_results]  # 仅保留必要回调
)

# 执行测试
test_trainer.test(model=model, datamodule=data_module)