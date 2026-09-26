"""One-off verification: fresh DB upgrade to head + constraint introspection."""
import os, sqlite3, tempfile, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.knowledge.models import CommitModel, FileChangeModel

t = CommitModel.__table__
print("commits uniques:", [(u.name, [c.name for c in u.columns]) for u in t.constraints if type(u).__name__ == "UniqueConstraint"])
f = FileChangeModel.__table__
print("file_changes fk constraints:", [
    (fc.name, [c.name for c in fc.columns], fc.referred_table.name,
     [rc.name for rc in fc.elements])
    for fc in f.constraints if type(fc).__name__ == "ForeignKeyConstraint"
])

# Fresh SQLite DB, full alembic upgrade to head
db = os.path.join(tempfile.gettempdir(), "_repointel_fresh_verify.db")
if os.path.exists(db):
    os.remove(db)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///" + db

from packages.knowledge import migrations
migrations.upgrade_to_head()
print("current == head:", migrations.current_revision() == migrations.head_revision())

c = sqlite3.connect(db)
print("alembic_version:", c.execute("SELECT version_num FROM alembic_version").fetchone())
print("--- commits DDL ---")
print(c.execute("SELECT sql FROM sqlite_master WHERE name='commits'").fetchone()[0])
print("--- file_changes DDL ---")
print(c.execute("SELECT sql FROM sqlite_master WHERE name='file_changes'").fetchone()[0])
print("--- commits indexes ---")
for r in c.execute("SELECT name, sql FROM sqlite_master WHERE tbl_name='commits' AND type='index'"):
    print(r)
os.remove(db)
print("OK")

