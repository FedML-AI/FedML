
def platform_is_valid(platform_type):
    if platform_type != 'octopus' and platform_type != 'parrot' \
            and platform_type != 'spider' and platform_type != 'beehive' and platform_type != 'cheetah' \
            and platform_type != 'falcon' and platform_type != 'launch':
        print("The platform should be the following options: {}".format(get_platform_options()))
        return False

    return True


def get_platform_options():
    return "octopus, parrot, spider, beehive, cheetah,falcon,launch"
