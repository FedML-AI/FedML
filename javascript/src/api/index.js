import axios from 'axios'

export function fetchConfig(params) {
    axios.post('https://jsonplaceholder.typicode.com/posts', params)
    .then((response) => {
        console.log(response.data);
    })
    .catch((error) => {
        console.error(error);
    });
}