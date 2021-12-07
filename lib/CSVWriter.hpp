//
// Created by Varun on 9/23/2017.
// https://thispointer.com/how-to-write-data-in-a-csv-file-in-c/
// Remy van Elst on 6/16/2019
// https://raymii.org/s/snippets/Cpp_create_and_write_to_a_csv_file.html
//

#ifndef DELPHI_CSVWRITER_HPP
#define DELPHI_CSVWRITER_HPP


#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <iterator>
#include <string>
#include <algorithm>
#include <mutex>

/*
 * A class to create and write data in a csv file.
 */
class CSVWriter
    {
  std::string fileName;
  std::string delimiter;
  int linesCount;
//  std::mutex logMutex;
      public:
      CSVWriter(std::string filename = "ag_timing.csv", std::string delm = ",") :
      fileName(filename), delimiter(delm), linesCount(0)
      {}
      /*
       * Member function to store a range as comma seperated value
       */
      template<typename T>
      void write_row(T first, T last);
    };
/*
 * This Function accepts a range and appends all the elements in the range
 * to the last row, separated by delimiter (Default is comma)
 */
template<typename T>
void CSVWriter::write_row(T first, T last)
{
//  std::lock_guard<std::mutex> csvLock(logMutex);
  std::fstream file;
  // Open the file in truncate mode if first line else in Append Mode
  file.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
  // Iterate over the range and add each element to file separated by delimiter.
  for (; first != last; )
  {
    file << *first;
    if (++first != last)
      file << this->delimiter;
  }
  file << "\n";
  linesCount++;
  // Close the file
  file.close();
};

#endif // DELPHI_CSVWRITER_HPP
