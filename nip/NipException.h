#ifndef NIPEXCEPTION_H
#define NIPEXCEPTION_H
#include "itkMacro.h"
#include "itkExceptionObject.h"
//
//
//
namespace MAC_nip
{
  /** \class NipException
   *
   * \brief Base exception class for classification conflicts.
   * 
   */
  class NipException : public itk::ExceptionObject
  {
  public:
    /** Run-time information. */
    itkTypeMacro(ImageFileReaderException, ExceptionObject);
    /** Constructor. */
    NipException( const char *file, unsigned int line,
		   const char *message = "Error in Nip",
		   const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Constructor. */
    NipException( const std::string & file, unsigned int line,
		   const char *message = "Error in Nip",
		   const char *loc = "Unknown" ):
    ExceptionObject( file, line, message, loc ){}
    /** Has to have empty throw(). */
    virtual ~NipException() throw() {};
  };
}
#endif
