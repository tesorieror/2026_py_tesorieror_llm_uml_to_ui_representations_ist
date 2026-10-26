-- Generated automatically from UML model uml10.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "project" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "description" CHARACTER VARYING(255) NOT NULL
  , "end_date" TIMESTAMP
  , "start_date" TIMESTAMP NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "employee" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "comment" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "date" TIMESTAMP NOT NULL
  , "text" CHARACTER VARYING(255) NOT NULL
  , "task_id" BIGINT NOT NULL
);

CREATE TABLE "task" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('ToDo', 'InProgress', 'Done', 'Blocked'))
  , "title" CHARACTER VARYING(255) NOT NULL
  , "due_date" TIMESTAMP NOT NULL
  , "priority" CHARACTER VARYING(255) NOT NULL CHECK ("priority" IN ('Low', 'Medium', 'High'))
  , "description" CHARACTER VARYING(255) NOT NULL
  , "project_id" BIGINT NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations
CREATE TABLE "employee_task" (
    "employee_id" CHARACTER VARYING(255) NOT NULL
  , "task_id" BIGINT NOT NULL
  , PRIMARY KEY ("employee_id", "task_id")
  , CONSTRAINT "fk_employee_task_employee"
        FOREIGN KEY ("employee_id")
        REFERENCES "employee" ("id")
  , CONSTRAINT "fk_employee_task_task"
        FOREIGN KEY ("task_id")
        REFERENCES "task" ("id")
);



-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "comment"
    ADD CONSTRAINT "fk_comment_task"
    FOREIGN KEY ("task_id")
    REFERENCES "task" ("id");


ALTER TABLE "task"
    ADD CONSTRAINT "fk_task_project"
    FOREIGN KEY ("project_id")
    REFERENCES "project" ("id");


