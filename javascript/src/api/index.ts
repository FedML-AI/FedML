export function fetchConfig(params: any) {
  return fetch('https://open.fedml.ai/fedmlOpsServer/configs/fetch', {
    method: 'post',
    body: JSON.stringify(params),
    headers: {
      'content-type': 'application/json',
    },
  })
    .then((res) => res.json())
    .then((res) => res.data)
    .catch((error) => console.error(error));
}
