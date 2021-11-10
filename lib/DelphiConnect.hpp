#pragma once

#include <sqlite3.h>

namespace DelphiConnect {

  sqlite3* delphiConnect(int mode);
  sqlite3* delphiConnectReadOnly();
  sqlite3* delphiConnectReadWrite();

}
