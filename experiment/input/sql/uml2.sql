-- Generated automatically from UML model uml2.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "customer" (
    "id" INTEGER PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "vip" BOOLEAN NOT NULL
);

CREATE TABLE "order" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "date" TIMESTAMP NOT NULL
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('Pending', 'Paid', 'Cancelled'))
  , "total" REAL NOT NULL
  , "customer_id" INTEGER NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "order"
    ADD CONSTRAINT "fk_order_customer"
    FOREIGN KEY ("customer_id")
    REFERENCES "customer" ("id");


