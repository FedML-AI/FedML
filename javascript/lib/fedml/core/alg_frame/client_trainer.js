export class ClientTrainer {
  constructor(model, args) {
    this.model = model
    this.id = 0
    this.args = args
  }

  set_id(trainer_id) {
    this.id = trainer_id
  }
}
