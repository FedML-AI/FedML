def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_lidc_idri", "fed-lidc-idri"]:
        from .lidc_idri_trainer import LIDCTrainer

        trainer = LIDCTrainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
