def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_camelyon16", "fed-camelyon16"]:
        from .camelyon16_trainer import Camelyon16Trainer

        trainer = Camelyon16Trainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
