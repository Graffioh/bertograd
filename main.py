from value import Value
from neuron import Neuron
from layer import Layer

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
    num_epochs = 10
    for i in range(num_epochs):
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

        # update to minimize r
        a = a - lr * a.grad
        b = b - lr * b.grad


    a.label = "a"
    b.label = "b"

    # this is not updating so no optimization. it's used just to print the final gradients correctly
    zero_grad((a,b))
    r.grad = 1
    r.backward()

    print("*" * 20)
    print("AFTER OPTIMIZATION:")
    print_values((a,b,y,z,t,r))

    print("*" * 20)
    n1 = Neuron(num_inputs=3)
    n1_out = n1.fire(inputs=(2,-3,8.5))
    print("NEURON 1 FIRED, RESULT=", n1_out)

    print("*" * 20)
    l1 = Layer(size=4, num_inputs=3)
    l1_out = l1.fire(inputs=[1.0, 0.5, -1.2])
    print("LAYER 1 FIRED, RESULT=", l1_out)

if __name__ == "__main__":
    main()