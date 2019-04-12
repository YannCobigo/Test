#ifndef EXCEPTION_H
#define EXCEPTION_H
#include "itkMacro.h"
#include "itkExceptionObject.h"
//
//
//
namespace NeuroBayes
{
  /** \class Exception
   *
   * \brief Base exception class for classification conflicts.
   * 
   */
  class NeuroBayesException : public itk::ExceptionObject
  {
  public:
    /** Run-time information. */
    itkTypeMacro(ImageFileReaderException, ExceptionObject);
    /** Constructor. */
    NeuroBayesException( const char *file, unsigned int line,
	       const char *message = "Error in NeuroBayes",
	       const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Constructor. */
    NeuroBayesException( const std::string & file, unsigned int line,
	       const char *message = "Error in NeuroBayes",
	       const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Has to have empty throw(). */
    virtual ~NeuroBayesException() throw() {};
  };
}
#endif
