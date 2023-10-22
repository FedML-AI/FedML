
def validate_platform(platform_type):
    if platform_type != 'octopus' and platform_type != 'parrot' \
            and platform_type != 'spider' and platform_type != 'beehive' and platform_type != 'cheetah' \
            and platform_type != 'falcon' and platform_type != 'launch':
        raise Exception("The platform should be the following options: {}".format(get_platform_options()))


def get_platform_options():
    return "octopus, parrot, spider, beehive, cheetah,falcon,launch"
