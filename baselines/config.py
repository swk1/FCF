class config:
    Config = {
        'current_directory': "/root/JIT-BiCC-main/",
    }
    
    @classmethod
    def get_config(cls):
        return cls.Config
    
    @classmethod
    def get_current_directory(cls):
        return cls.Config['current_directory']
