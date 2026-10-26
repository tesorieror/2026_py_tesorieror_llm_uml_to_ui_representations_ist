-- Generated automatically from UML model uml9.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "product" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "description" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
  , "price" REAL NOT NULL
  , "sku" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "stock_movement" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "quantity" REAL NOT NULL
  , "date" TIMESTAMP NOT NULL
  , "type" CHARACTER VARYING(255) NOT NULL CHECK ("type" IN ('Inbound', 'Outbound', 'Adjustment'))
  , "stock_item_id" BIGINT NOT NULL
);

CREATE TABLE "stock_item" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "quantity" INTEGER NOT NULL
  , "minimum_quantity" INTEGER NOT NULL
  , "warehouse_id" BIGINT NOT NULL
  , "product_id" BIGINT NOT NULL
);

CREATE TABLE "warehouse" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "location" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
  , "code" CHARACTER VARYING(255) NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "stock_movement"
    ADD CONSTRAINT "fk_stock_movement_stock_item"
    FOREIGN KEY ("stock_item_id")
    REFERENCES "stock_item" ("id");


ALTER TABLE "stock_item"
    ADD CONSTRAINT "fk_stock_item_warehouse"
    FOREIGN KEY ("warehouse_id")
    REFERENCES "warehouse" ("id");


ALTER TABLE "stock_item"
    ADD CONSTRAINT "fk_stock_item_product"
    FOREIGN KEY ("product_id")
    REFERENCES "product" ("id");


