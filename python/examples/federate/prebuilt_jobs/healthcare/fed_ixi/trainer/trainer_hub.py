def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_ixi", "fed-ixi"]:
        from .ixi_trainer import IXITrainer

        trainer = IXITrainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
