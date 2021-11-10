#include "DelphiConnect.hpp"
#include <sqlite3.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

namespace DelphiConnect {

  sqlite3* delphiConnect(int mode) {
    char* pPath = getenv ("DELPHI_DB");
    if (pPath == NULL) {
      cout << "\n\nERROR: DELPHI_DB environment variable containing the path to delphi.db is not set!\n\n";
      exit(1);
    }

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(getenv("DELPHI_DB"), &db, mode, NULL) != SQLITE_OK) {
      cout << "\n\nERROR: delphi.db does not exist at " << pPath << endl;
      cout << sqlite3_errmsg(db) << endl;
      exit(1);
    }

    return db;
  }

  sqlite3* delphiConnectReadOnly() {
    return delphiConnect(SQLITE_OPEN_READONLY);
  }

  sqlite3* delphiConnectReadWrite() {
    return delphiConnect(SQLITE_OPEN_READWRITE);
  }
}
