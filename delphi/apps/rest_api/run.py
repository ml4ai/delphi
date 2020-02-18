# Run a test server.
from delphi.paths import db_path
from delphi.apps.rest_api import create_app

def main():
    app = create_app(debug=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
