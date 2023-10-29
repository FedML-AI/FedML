def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_kits19", "fed_kits2019", "fed-kits2019", "fed-kits19"]:
        from .kits19_trainer import KITS19Trainer

        trainer = KITS19Trainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
