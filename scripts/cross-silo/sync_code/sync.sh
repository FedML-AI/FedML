#!/bin/bash
set -x
#sh async_source_code_to_single_node.sh trpc-server
sh async_source_code_to_single_node.sh trpc-client0
sh async_source_code_to_single_node.sh trpc-client1
sh async_source_code_to_single_node.sh trpc-client2
sh async_source_code_to_single_node.sh trpc-client3
sh async_source_code_to_single_node.sh trpc-client4
sh async_source_code_to_single_node.sh trpc-client5
sh async_source_code_to_single_node.sh trpc-client6
sh async_source_code_to_single_node.sh trpc-client7