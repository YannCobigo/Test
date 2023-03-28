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
#ifndef MYCLASS_H
#define MYCLASS_H

namespace my_namespace {

/**
 * @file MyClass.hpp
 * @brief Header file for the MyClass class.
 *
 * This is a more detailed description of MyClass, which can span
 * multiple lines.
 *
 * @author Your Name
 * @date 2023-03-20
 *
 * @license MIT License
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
