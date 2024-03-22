# timeout = 0.5, each will cost 0.1s
# we will send 10 requests in total, and see how many requests will be timeout, cancelled by server

# curl -XPOST localhost:2345/predict -d '{"text": "Hello", "stream": true}'

import asyncio
import aiohttp
import time

parallel_req_num = 10
round_num = 1

FAILED_CNT = 0
TIMEOUT_CNT = 0


async def time_consume_operation():
    global FAILED_CNT
    global TIMEOUT_CNT

    print("start req")
    timeout = aiohttp.ClientTimeout(total=3)
    response = None
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("http://localhost:2345/predict", headers={}, json={
                "stream": True
            }) as response:
                # print(response.status)
                # print(await response.text())
                pass
    except Exception as e:
        TIMEOUT_CNT += 1

    if response is not None and response.status != 200:
        FAILED_CNT += 1

    print("end req")


# for i in range(parallel_req_num):
#     print(f"Start {i}th request")
#     asyncio.run(time_consume_operation())

async def main():
    tasks = []

    for _ in range(parallel_req_num):
        task = asyncio.create_task(time_consume_operation())
        tasks.append(task)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    start_time = time.time()
    for i in range(round_num):
        print(f"Start {i+1}th round")
        asyncio.run(main())
        print(f"End {i+1}th round")

    print(f"Total time: {time.time() - start_time}, avg latency:"
          f" {(time.time() - start_time) / (parallel_req_num * round_num)}")
    print("All requests are done, total try:", parallel_req_num * round_num,
          "failed cnt:", FAILED_CNT, "timeout cnt:", TIMEOUT_CNT)


