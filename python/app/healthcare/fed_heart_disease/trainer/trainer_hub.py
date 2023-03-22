def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_heart_disease", "fed-heart-disease"]:
        from .heart_disease_trainer import HeartDiseaseTrainer

        trainer = HeartDiseaseTrainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
