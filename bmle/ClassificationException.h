#ifndef BMLEEXCEPTION_H
#define BMLEEXCEPTION_H
#include "itkMacro.h"
#include "itkExceptionObject.h"
//
//
//
namespace MAC_bmle
{
  /** \class BmleException
   *
   * \brief Base exception class for classification conflicts.
   * 
   */
  class BmleException : public itk::ExceptionObject
  {
  public:
    /** Run-time information. */
    itkTypeMacro(ImageFileReaderException, ExceptionObject);
    /** Constructor. */
    BmleException( const char *file, unsigned int line,
			     const char *message = "Error in Bmle",
			     const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Constructor. */
    BmleException( const std::string & file, unsigned int line,
			     const char *message = "Error in Bmle",
			     const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Has to have empty throw(). */
    virtual ~BmleException() throw() {};
  };
}
#endif
