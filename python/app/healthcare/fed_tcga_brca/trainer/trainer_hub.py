def create_trainer(model, args):
    trainer_name = str(args.dataset).lower()
    if trainer_name in ["fed_tcga_brca", "fed-tcga-brca"]:
        from .tcga_brca_trainer import TcgaBrcaTrainer

        trainer = TcgaBrcaTrainer(model=model, args=args)
    else:
        raise ValueError(f"Trainer {trainer_name} not implemented")

    return trainer
