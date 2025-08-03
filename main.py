from value import Value

def print_values(values):
    for v in values:
        print(f"{v}")

def zero_grad(values):
    for v in values:
        v.grad = 0.0

def main():
    a = Value(10)
    b = Value(20)

    lr = 0.01
    for i in range(20):
        y = a*b
        y.label = "y"

        z = b+b 
        z.label = "z"

        t = y+z
        t.label = "t"

        r = t*2
        r.label = "r"


        # backprop
        zero_grad((a,b))
        r.grad = 1
        r.backward()

        if i == 0:
            print("FIRST COMPUTATION:")
            print_values((a,b,y,z,t,r))

        # update
        a = a - lr * a.grad
        b = b - lr * b.grad


    a.label = "a"
    b.label = "b"

    # this is not updating so no optimization, is just to print the final gradients
    zero_grad((a,b))
    r.grad = 1
    r.backward()

    print("*" * 20)
    print("AFTER OPTIMIZATION:")
    print_values((a,b,y,z,t,r))

if __name__ == "__main__":
    main()