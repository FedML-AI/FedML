import {constant} from '../constants'

export class CrossSiloLauncher {
  static launch_dist_trainers(torch_client_filename, inputs){
    const args = this.load_arguments(constant.FEDML_TRAINING_PLATFORM_CROSS_SILO)
  }

  static load_arguments(training_type=null, comm_backend=null){
    const cmd_args = add_args()
  }

  
}