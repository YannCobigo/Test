#ifndef MYCLASS_H
#define MYCLASS_H

namespace my_namespace {

/**
 * @brief This is a brief description of MyClass.
 *
 * This is a more detailed description of MyClass, which can span
 * multiple lines.
 */
class MyClass {
public:
    /**
     * @brief Default constructor for MyClass.
     */
    MyClass();

    /**
     * @brief Destructor for MyClass.
     */
    ~MyClass();

    /**
     * @brief Add two integers.
     *
     * This function takes two integers as input and returns their sum.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return The sum of a and b.
     */
    int add(int a, int b);

    /**
     * @brief Subtract two integers.
     *
     * This function takes two integers as input and returns their difference.
     *
     * @param a First integer.
     * @param b Second integer.
     * @return The difference of a and b.
     */
    int subtract(int a, int b);
};

} // end namespace my_namespace

#endif // MYCLASS_H
