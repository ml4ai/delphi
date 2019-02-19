# Run a test server.
from delphi.paths import db_path
from delphi.icm_api import create_app

if __name__ == "__main__":
    app = create_app(debug=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.run(host="127.0.0.1", port=5000)
