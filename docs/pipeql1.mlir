module {
  func.func @main() {
    %0 = relalg.query (){
      %1 = relalg.basetable  {table_identifier = "users"} columns: {age => @users::@age({type = !db.nullable<i32>}), email => @users::@email({type = !db.nullable<!db.string>}), id => @users::@id({type = !db.nullable<i32>}), name => @users::@name({type = !db.string})}
      %2 = relalg.materialize %1 [@users::@id,@users::@name,@users::@age,@users::@email] => ["id", "name", "age", "email"] : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
      relalg.query_return %2 : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    } -> !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    subop.set_result 0 %0 : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    return
  }
}