class UserNotFoundException(Exception):
    def __init__(self, message="Usuario no existente"):
        self.message = message

        super().__init__(self.message)