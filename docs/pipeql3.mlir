module {
  func.func @main() {
    %0 = relalg.query (){
      %1 = relalg.basetable  {table_identifier = "users"} columns: {age => @users::@age({type = !db.nullable<i32>}), email => @users::@email({type = !db.nullable<!db.string>}), id => @users::@id({type = !db.nullable<i32>}), name => @users::@name({type = !db.string})}
      %2 = relalg.selection %1 (%arg0: !tuples.tuple){
        %4 = tuples.getcol %arg0 @users::@age : !db.nullable<i32>
        %5 = db.constant(18 : i32) : i32
        %6 = db.compare gt %4 : !db.nullable<i32>, %5 : i32
        tuples.return %6 : !db.nullable<i1>
      }
      %3 = relalg.materialize %2 [@users::@id,@users::@name,@users::@age,@users::@email] => ["id", "name", "age", "email"] : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
      relalg.query_return %3 : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    } -> !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    subop.set_result 0 %0 : !subop.local_table<[id$0 : !db.nullable<i32>, name$0 : !db.string, age$0 : !db.nullable<i32>, email$0 : !db.nullable<!db.string>], ["id", "name", "age", "email"]>
    return
  }
}