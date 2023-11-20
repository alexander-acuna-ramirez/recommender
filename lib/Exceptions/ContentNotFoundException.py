class ContentNotFoundException(Exception):
    def __init__(self, message="Contenido no existente"):
        self.message = message

        super().__init__(self.message)