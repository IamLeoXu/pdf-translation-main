class ScannedPDFError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ExtractTextError(Exception):
    def __init__(self, message):
        super().__init__(message)


class TranslationFailedError(Exception):
    """One or more attempted paragraphs have no accepted translation."""



class InputFileGeneratedByBabelDOCError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ContentFilterError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
