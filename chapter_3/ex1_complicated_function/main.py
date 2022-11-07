import traceback
import sys

def convert_to_dtype(*args, dtype, debug=False):
    result = []
    for index, arg in enumerate(args):
        try:
            result.append(dtype(arg))
        except ValueError as e:
            exc_info = sys.exc_info()
            if debug:
                print(f"ErrorMessage: Can't convert '{arg}' at index {index} # this is from the debug-flag")
            else:
                pass
            print(traceback.format_exc())
            break
    if len(result) == len([*args]):
        return result


if __name__ == "__main__":
    print(convert_to_dtype("1", 4, 5.0, dtype=int))
    # [1, 4, 5]

    print(convert_to_dtype((1,0), "a", 15.1516, dtype=str))
    # [’(1, 0)’, ’a’, ’15.1516’]


    print(convert_to_dtype(5, "a", dtype=int, debug=False))
    # Traceback
    # ValueError: invalid literal for int() with base 10: ’a

    print(convert_to_dtype(5, "a", dtype=int, debug=True))
    # ErrorMessage: Can’t convert ’a’ at index 1 # this is from the debug-flag
    # Traceback
    # ValueError: invalid literal for int() with base 10: ’a'



