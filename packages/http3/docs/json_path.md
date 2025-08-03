# RFC 9535: JSONPath Query Expressions for JSON

**Published:** February 2024  
**Status:** IETF Standards Track  
**Editors:** S. Gössner, G. Normington, C. Bormann

## Abstract

JSONPath defines a string syntax for selecting and extracting JSON (RFC 8259) values from within a given JSON value.

## 1. Introduction

JSON [RFC8259] is a popular representation format for structured data values. JSONPath defines a string syntax for selecting and extracting JSON values from within a given JSON value.

In relation to JSON Pointer [RFC6901], JSONPath is not intended as a replacement but as a more powerful companion.

### 1.1 Terminology

**Key Terms:**
- **Value**: Data item conforming to the generic data model of JSON (primitive or structured data)
- **Member**: A name/value pair in an object
- **Name**: The name (string) in a name/value pair constituting a member
- **Element**: A value in a JSON array
- **Index**: An integer that identifies a specific element in an array
- **Query**: Short name for a JSONPath expression
- **Query Argument**: The value a JSONPath expression is applied to
- **Location**: The position of a value within the query argument

#### 1.1.1 JSON Values as Trees of Nodes

A JSON value can be thought of as a tree structure where:
- Each value in the JSON (including root) is a node
- Values nested within structured values are child nodes
- The root of the tree is the query argument itself

### 1.4 Overview of JSONPath Expressions

A JSONPath expression is a string that:
1. Begins with a root identifier `$`
2. Is followed by zero or more segments
3. Each segment is either child segment `[<selectors>]` or descendant segment `..[<selectors>]`

#### 1.4.1 Identifiers
- **Root identifier** `$`: Refers to the root node of the query argument

#### 1.4.2 Segments
- **Child segment** `[<selectors>]`: Selects zero or more children of a node
- **Descendant segment** `..[<selectors>]`: Selects zero or more descendants of a node

#### 1.4.3 Selectors
- **Name selector** `'name'` or `"name"`: Selects a named child of an object
- **Wildcard selector** `*`: Selects all children of a node
- **Index selector** `0`, `1`, `-1`: Selects an indexed child of an array
- **Array slice selector** `1:3`, `1:`, `:3`: Selects array children specified by a slice
- **Filter selector** `?<logical-expr>`: Tests a logical expression against each child

### 1.5 JSONPath Examples

Given JSON:
```json
{
  "store": {
    "book": [
      {
        "category": "reference",
        "author": "Nigel Rees",
        "title": "Sayings of the Century",
        "price": 8.95
      },
      {
        "category": "fiction",
        "author": "Evelyn Waugh", 
        "title": "Sword of Honour",
        "price": 12.99
      }
    ],
    "bicycle": {
      "color": "red",
      "price": 19.95
    }
  }
}
```

Example queries:
- `$.store.book[*].author` → All book authors
- `$..author` → All authors 
- `$.store.*` → All things in store
- `$.store..price` → All prices in store
- `$..book[2]` → Third book
- `$..book[-1]` → Last book  
- `$..book[0,1]` → First two books
- `$..book[:2]` → First two books
- `$..book[?@.isbn]` → Books with ISBN
- `$..book[?@.price<10]` → Books cheaper than 10

## 2. JSONPath Syntax and Semantics

### 2.1 Overview

#### 2.1.1 Syntax

```abnf
jsonpath-query      = root-identifier *segment
root-identifier     = "$"
segment             = child-segment / descendant-segment
child-segment       = "[" selector-list "]"
descendant-segment  = ".." child-segment
selector-list       = selector *("," selector)
selector            = name-selector / wildcard-selector / 
                      index-selector / array-slice-selector / 
                      filter-selector
```

#### 2.1.2 Semantics

A JSONPath query is applied to a query argument and produces a nodelist (list of zero or more nodes from the query argument).

### 2.2 Root Identifier

#### 2.2.1 Syntax
```abnf
root-identifier = "$"
```

#### 2.2.2 Semantics
The root identifier `$` refers to the root node of the query argument.

### 2.3 Selectors

#### 2.3.1 Name Selector

##### 2.3.1.1 Syntax
```abnf
name-selector = string-literal
string-literal = %x22 *double-quoted %x22 / %x27 *single-quoted %x27
double-quoted = unescaped / %x5C escaped
single-quoted = unescaped / %x5C escaped  
```

##### 2.3.1.2 Semantics
A name selector `'name'` selects at most one object member value. The member is selected if the object has a member with name equal to the selector's string value.

##### 2.3.1.3 Examples
```jsonpath
$['store']
$["store"]
```

#### 2.3.2 Wildcard Selector

##### 2.3.2.1 Syntax
```abnf
wildcard-selector = "*"
```

##### 2.3.2.2 Semantics
A wildcard selector `*` selects all children of a node:
- For objects: selects the values of all members
- For arrays: selects all elements

##### 2.3.2.3 Examples
```jsonpath
$[*]        ; All children of root
$.store[*]  ; All children of store
```

#### 2.3.3 Index Selector

##### 2.3.3.1 Syntax
```abnf
index-selector = int
int = "0" / (["-"] %x31-39 *%x30-39)
```

##### 2.3.3.2 Semantics
An index selector `<int>` selects at most one array element value. The element is selected if the array has an element at the specified index.

- Non-negative indices select from the start (0-based)
- Negative indices select from the end (-1 is last element)

##### 2.3.3.3 Examples
```jsonpath
$[0]    ; First array element  
$[-1]   ; Last array element
$[2]    ; Third array element
```

#### 2.3.4 Array Slice Selector

##### 2.3.4.1 Syntax
```abnf
array-slice-selector = [start] ":" [end] [":" [step]]
start = int
end = int  
step = int
```

##### 2.3.4.2 Semantics
An array slice selector `<start>:<end>:<step>` selects elements from arrays in a specified range:
- `start`: Starting index (inclusive)
- `end`: Ending index (exclusive)  
- `step`: Step size (default 1)

##### 2.3.4.3 Examples
```jsonpath
$[1:3]     ; Elements 1 and 2
$[1:]      ; Elements from 1 to end
$[:3]      ; Elements 0, 1, 2
$[::2]     ; Every second element
$[1:5:2]   ; Elements 1, 3 (step of 2)
```

#### 2.3.5 Filter Selector

##### 2.3.5.1 Syntax
```abnf
filter-selector = "?" logical-expr
logical-expr = logical-or-expr
logical-or-expr = logical-and-expr *("||" logical-and-expr)
logical-and-expr = basic-expr *("&&" basic-expr)
basic-expr = paren-expr / comparison-expr / test-expr
comparison-expr = comparable "==" comparable / 
                  comparable "!=" comparable /
                  comparable "<=" comparable /
                  comparable "<" comparable /
                  comparable ">=" comparable /
                  comparable ">" comparable
test-expr = filter-query / function-expr
```

##### 2.3.5.2 Semantics
A filter selector `?<logical-expr>` selects nodes for which the logical expression evaluates to true.

The current node being tested is referenced by `@`.

##### 2.3.5.3 Examples
```jsonpath
$[?@.price < 10]           ; Nodes with price less than 10
$[?@.category == 'fiction'] ; Nodes with category 'fiction'
$[?@.isbn]                 ; Nodes that have an isbn member
```

### 2.4 Function Extensions

JSONPath includes built-in function extensions:

#### 2.4.4 length() Function Extension
Returns the length of a value:
- String: number of characters
- Array: number of elements  
- Object: number of members
- `null`: returns `null`

#### 2.4.5 count() Function Extension
Returns the number of nodes in a nodelist.

#### 2.4.6 match() Function Extension
Tests if a string matches a regular expression.

#### 2.4.7 search() Function Extension  
Tests if a string contains a match for a regular expression.

#### 2.4.8 value() Function Extension
Converts a single-node nodelist to the value of its node.

### 2.5 Segments

#### 2.5.1 Child Segment

##### 2.5.1.1 Syntax
```abnf
child-segment = "[" selector-list "]"
```

##### 2.5.1.2 Semantics
A child segment `[<selectors>]` produces a nodelist consisting of child nodes of the input value selected by each selector.

#### 2.5.2 Descendant Segment  

##### 2.5.2.1 Syntax
```abnf
descendant-segment = ".." child-segment
```

##### 2.5.2.2 Semantics
A descendant segment `..[<selectors>]` produces a nodelist consisting of descendant nodes of the input value selected by each selector.

## 3. IANA Considerations

### 3.1 Registration of Media Type application/jsonpath

A new media type `application/jsonpath` has been registered for JSONPath expressions.

## Key Implementation Requirements

1. **Root Identifier**: All JSONPath expressions MUST begin with `$`
2. **Selector Syntax**: Must support all five selector types (name, wildcard, index, slice, filter)
3. **Segment Types**: Must support both child `[...]` and descendant `..[...]` segments
4. **Function Extensions**: Should support the five built-in functions
5. **Unicode Support**: Must handle Unicode characters properly in string literals
6. **Error Handling**: Must handle invalid queries gracefully

## Compliance with RFC 9535

This specification provides the authoritative definition of JSONPath syntax and semantics. Implementations should:

1. Follow the ABNF grammar precisely
2. Implement all required selectors and segments
3. Handle edge cases as specified
4. Support the defined function extensions
5. Provide appropriate error messages for invalid queries

---

**Note**: This markdown summary covers the key aspects of RFC 9535. For complete technical details, refer to the full specification text in `rfc9535.txt`.