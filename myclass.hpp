// MIT License
//
// Copyright (c) [year] [author]
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
/**
 * @file myclass.hpp
 * @brief This file defines the MyClass class.
 *
 * @author Your Name
 * @date 2023-03-20
 *
 * @license MIT License
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
