
def platform_is_valid(platform_type):
    if platform_type != 'octopus' and platform_type != 'parrot' \
            and platform_type != 'spider' and platform_type != 'beehive':
        print("The platform should be the following options: octopus, parrot, spider, beehive")
        return False

    return True
