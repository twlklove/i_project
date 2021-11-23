from hello_pb2 import helloworld


def main():
    hw = helloworld()
    with open("mybuffer.io", "rb") as f:
        content = f.read()
        hw.ParseFromString(content)
        print(hw.id)
        print(hw.str)


if __name__ == "__main__":
    main()

