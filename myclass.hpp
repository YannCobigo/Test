/**
 * @file myclass.hpp
 * @brief This file defines the MyClass class.
 */

#ifndef MYCLASS_HPP
#define MYCLASS_HPP

/**
 * @class MyClass
 * @brief This class represents a simple example class.
 *
 * The MyClass class provides a basic example of a C++ class with some simple member variables and methods.
 */
class MyClass {
public:
    /**
     * @brief Constructs a new instance of the MyClass class.
     *
     * This constructor initializes the member variables of the MyClass class to their default values.
     */
    MyClass();

    /**
     * @brief Destructs an instance of the MyClass class.
     *
     * This destructor frees any resources allocated by the MyClass class.
     */
    ~MyClass();

    /**
     * @brief Gets the value of the x member variable.
     *
     * This method returns the current value of the x member variable.
     *
     * @return The current value of the x member variable.
     */
    int getX() const;

    /**
     * @brief Sets the value of the x member variable.
     *
     * This method sets the value of the x member variable to the specified value.
     *
     * @param[in] x The new value of the x member variable.
     */
    void setX(int x);

    /**
     * @brief Gets the value of the y member variable.
     *
     * This method returns the current value of the y member variable.
     *
     * @return The current value of the y member variable.
     */
    int getY() const;

    /**
     * @brief Sets the value of the y member variable.
     *
     * This method sets the value of the y member variable to the specified value.
     *
     * @param[in] y The new value of the y member variable.
     */
    void setY(int y);

private:
    int x; /**< The x coordinate of the MyClass instance. */
    int y; /**< The y coordinate of the MyClass instance. */
};

#endif // MYCLASS_HPP
