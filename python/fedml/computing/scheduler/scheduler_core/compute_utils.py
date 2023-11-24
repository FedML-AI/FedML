

class ComputeUitls(object):

    @staticmethod
    def map_list_to_str(list_obj):
        list_map = map(lambda x: str(x), list_obj[0:])
        list_str = ",".join(list_map)
        return list_str

    @staticmethod
    def map_str_list_to_int_list(list_obj):
        list_map = map(lambda x: int(x), list_obj[0:])
        return list(list_map)


