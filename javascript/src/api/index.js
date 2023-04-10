import axios from 'axios'

export function fetchConfig(params) {
    axios.post('https://open.fedml.ai/fedmlOpsServer/configs/fetch', params)
    .then((response) => {
        console.log(response.data);
    })
    .catch((error) => {
        console.error(error);
    });
}