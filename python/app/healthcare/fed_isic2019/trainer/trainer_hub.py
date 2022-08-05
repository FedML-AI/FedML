def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_isic2019", "fed-isic2019"]:
        from .isic2019_trainer import ISIC2019Trainer

        trainer = ISIC2019Trainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
