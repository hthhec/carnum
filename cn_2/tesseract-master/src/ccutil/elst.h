/**********************************************************************
 * File:        elst.h  (Formerly elist.h)
 * Description: Embedded list module include file.
 * Author:      Phil Cheatle
 *
 * (C) Copyright 1991, Hewlett-Packard Ltd.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 *
 **********************************************************************/

#ifndef ELST_H
#define ELST_H

#include "list.h"
#include "lsterr.h"
#include "serialis.h"

#include <cstdio>

namespace tesseract {

class ELIST_ITERATOR;

/**********************************************************************
This module implements list classes and iterators.
The following list types and iterators are provided:

  List type        List Class      Iterator Class     Element Class
  ---------         ----------      --------------      -------------

    Embedded list       ELIST
              ELIST_ITERATOR
              ELIST_LINK
    (Single linked)

    Embedded list       ELIST2
              ELIST2_ITERATOR
              ELIST2_LINK
    (Double linked)

    Cons List           CLIST
              CLIST_ITERATOR
              CLIST_LINK
    (Single linked)

An embedded list is where the list pointers are provided by a generic class.
Data types to be listed inherit from the generic class.  Data is thus linked
in only ONE list at any one time.

A cons list has a separate structure for a "cons cell".  This contains the
list pointer(s) AND a pointer to the data structure held on the list.  A
structure can be on many cons lists at the same time, and the structure does
not need to inherit from any generic class in order to be on the list.

The implementation of lists is very careful about space and speed overheads.
This is why many embedded lists are provided. The same concerns mean that
in-line type coercion is done, rather than use virtual functions.  This is
cumbersome in that each data type to be listed requires its own iterator and
list class - though macros can generate these.  It also prevents heterogeneous
lists.
**********************************************************************/

/**********************************************************************
 *                          CLASS - ELIST_LINK
 *
 *                          Generic link class for singly linked lists with
 *embedded links
 *
 *  Note:  No destructor - elements are assumed to be destroyed EITHER after
 *  they have been extracted from a list OR by the ELIST destructor which
 *  walks the list.
 **********************************************************************/

class ELIST_LINK {
  friend class ELIST_ITERATOR;
  friend class ELIST;

  ELIST_LINK *next;

public:
  ELIST_LINK() {
    next = nullptr;
  }
  // constructor

  // The special copy constructor is used by lots of classes.
  ELIST_LINK(const ELIST_LINK &) {
    next = nullptr;
  }

  // The special assignment operator is used by lots of classes.
  void operator=(const ELIST_LINK &) {
    next = nullptr;
  }
};

/**********************************************************************
 * CLASS - ELIST
 *
 * Generic list class for singly linked lists with embedded links
 **********************************************************************/

class TESS_API ELIST {
  friend class ELIST_ITERATOR;

  ELIST_LINK *last; // End of list
  //(Points to head)
  ELIST_LINK *First() { // return first
    return last ? last->next : nullptr;
  }

public:
  ELIST() { // constructor
    last = nullptr;
  }

  void internal_clear( // destroy all links
                       // ptr to zapper functn
      void (*zapper)(void *));

  bool empty() const { // is list empty?
    return !last;
  }

  bool singleton() const {
    return last ? (last == last->next) : false;
  }

  void shallow_copy(      // dangerous!!
      ELIST *from_list) { // beware destructors!!
    last = from_list->last;
  }

  // ptr to copier functn
  void internal_deep_copy(ELIST_LINK *(*copier)(ELIST_LINK *),
                          const ELIST *list); // list being copied

  void assign_to_sublist(       // to this list
      ELIST_ITERATOR *start_it, // from list start
      ELIST_ITERATOR *end_it);  // from list end

  // # elements in list
  int32_t length() const {
    int32_t count = 0;
    if (last != nullptr) {
      count = 1;
      for (auto it = last->next; it != last; it = it->next) {
        count++;
      }
    }
    return count;
  }

  void sort(          // sort elements
      int comparator( // comparison routine
          const void *, const void *));

  // Assuming list has been sorted already, insert new_link to
  // keep the list sorted according to the same comparison function.
  // Comparison function is the same as used by sort, i.e. uses double
  // indirection. Time is O(1) to add to beginning or end.
  // Time is linear to add pre-sorted items to an empty list.
  // If unique is set to true and comparator() returns 0 (an entry with the
  // same information as the one contained in new_link is already in the
  // list) - new_link is not added to the list and the function returns the
  // pointer to the identical entry that already exists in the list
  // (otherwise the function returns new_link).
  ELIST_LINK *add_sorted_and_find(int comparator(const void *, const void *), bool unique,
                                  ELIST_LINK *new_link);

  // Same as above, but returns true if the new entry was inserted, false
  // if the identical entry already existed in the list.
  bool add_sorted(int comparator(const void *, const void *), bool unique, ELIST_LINK *new_link) {
    return (add_sorted_and_find(comparator, unique, new_link) == new_link);
  }
};

/***********************************************************************
 *                          CLASS - ELIST_ITERATOR
 *
 *                          Generic iterator class for singly linked lists with
 *embedded links
 **********************************************************************/

class TESS_API ELIST_ITERATOR {
  friend void ELIST::assign_to_sublist(ELIST_ITERATOR *, ELIST_ITERATOR *);

  ELIST *list;                  // List being iterated
  ELIST_LINK *prev;             // prev element
  ELIST_LINK *current;          // current element
  ELIST_LINK *next;             // next element
  ELIST_LINK *cycle_pt;         // point we are cycling the list to.
  bool ex_current_was_last;     // current extracted was end of list
  bool ex_current_was_cycle_pt; // current extracted was cycle point
  bool started_cycling;         // Have we moved off the start?

  ELIST_LINK *extract_sublist(   // from this current...
      ELIST_ITERATOR *other_it); // to other current

public:
  ELIST_ITERATOR() { // constructor
    list = nullptr;
  } // unassigned list

  explicit ELIST_ITERATOR(ELIST *list_to_iterate);

  void set_to_list( // change list
      ELIST *list_to_iterate);

  void add_after_then_move(  // add after current &
      ELIST_LINK *new_link); // move to new

  void add_after_stay_put(   // add after current &
      ELIST_LINK *new_link); // stay at current

  void add_before_then_move( // add before current &
      ELIST_LINK *new_link); // move to new

  void add_before_stay_put(  // add before current &
      ELIST_LINK *new_link); // stay at current

  void add_list_after(     // add a list &
      ELIST *list_to_add); // stay at current

  void add_list_before(    // add a list &
      ELIST *list_to_add); // move to it 1st item

  ELIST_LINK *data() { // get current data
#ifndef NDEBUG
    if (!list) {
      NO_LIST.error("ELIST_ITERATOR::data", ABORT, nullptr);
    }
    if (!current) {
      NULL_DATA.error("ELIST_ITERATOR::data", ABORT, nullptr);
    }
#endif
    return current;
  }

  ELIST_LINK *data_relative( // get data + or - ...
      int8_t offset);        // offset from current

  ELIST_LINK *forward(); // move to next element

  ELIST_LINK *extract(); // remove from list

  ELIST_LINK *move_to_first(); // go to start of list

  ELIST_LINK *move_to_last(); // go to end of list

  void mark_cycle_pt(); // remember current

  bool empty() const { // is list empty?
#ifndef NDEBUG
    if (!list) {
      NO_LIST.error("ELIST_ITERATOR::empty", ABORT, nullptr);
    }
#endif
    return list->empty();
  }

  bool current_extracted() const { // current extracted?
    return !current;
  }

  bool at_first() const; // Current is first?

  bool at_last() const; // Current is last?

  bool cycled_list() const; // Completed a cycle?

  void add_to_end(           // add at end &
      ELIST_LINK *new_link); // don't move

  void exchange(                 // positions of 2 links
      ELIST_ITERATOR *other_it); // other iterator

  //# elements in list
  int32_t length() const {
    return list->length();
  }

  void sort(          // sort elements
      int comparator( // comparison routine
          const void *, const void *));
};

/***********************************************************************
 *                          ELIST_ITERATOR::set_to_list
 *
 *  (Re-)initialise the iterator to point to the start of the list_to_iterate
 *  over.
 **********************************************************************/

inline void ELIST_ITERATOR::set_to_list( // change list
    ELIST *list_to_iterate) {
#ifndef NDEBUG
  if (!list_to_iterate) {
    BAD_PARAMETER.error("ELIST_ITERATOR::set_to_list", ABORT, "list_to_iterate is nullptr");
  }
#endif

  list = list_to_iterate;
  prev = list->last;
  current = list->First();
  next = current ? current->next : nullptr;
  cycle_pt = nullptr; // await explicit set
  started_cycling = false;
  ex_current_was_last = false;
  ex_current_was_cycle_pt = false;
}

/***********************************************************************
 *                          ELIST_ITERATOR::ELIST_ITERATOR
 *
 *  CONSTRUCTOR - set iterator to specified list;
 **********************************************************************/

inline ELIST_ITERATOR::ELIST_ITERATOR(ELIST *list_to_iterate) {
  set_to_list(list_to_iterate);
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_after_then_move
 *
 *  Add a new element to the list after the current element and move the
 *  iterator to the new element.
 **********************************************************************/

inline void ELIST_ITERATOR::add_after_then_move( // element to add
    ELIST_LINK *new_element) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_after_then_move", ABORT, nullptr);
  }
  if (!new_element) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_after_then_move", ABORT, "new_element is nullptr");
  }
  if (new_element->next) {
    STILL_LINKED.error("ELIST_ITERATOR::add_after_then_move", ABORT, nullptr);
  }
#endif

  if (list->empty()) {
    new_element->next = new_element;
    list->last = new_element;
    prev = next = new_element;
  } else {
    new_element->next = next;

    if (current) { // not extracted
      current->next = new_element;
      prev = current;
      if (current == list->last) {
        list->last = new_element;
      }
    } else { // current extracted
      prev->next = new_element;
      if (ex_current_was_last) {
        list->last = new_element;
      }
      if (ex_current_was_cycle_pt) {
        cycle_pt = new_element;
      }
    }
  }
  current = new_element;
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_after_stay_put
 *
 *  Add a new element to the list after the current element but do not move
 *  the iterator to the new element.
 **********************************************************************/

inline void ELIST_ITERATOR::add_after_stay_put( // element to add
    ELIST_LINK *new_element) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_after_stay_put", ABORT, nullptr);
  }
  if (!new_element) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_after_stay_put", ABORT, "new_element is nullptr");
  }
  if (new_element->next) {
    STILL_LINKED.error("ELIST_ITERATOR::add_after_stay_put", ABORT, nullptr);
  }
#endif

  if (list->empty()) {
    new_element->next = new_element;
    list->last = new_element;
    prev = next = new_element;
    ex_current_was_last = false;
    current = nullptr;
  } else {
    new_element->next = next;

    if (current) { // not extracted
      current->next = new_element;
      if (prev == current) {
        prev = new_element;
      }
      if (current == list->last) {
        list->last = new_element;
      }
    } else { // current extracted
      prev->next = new_element;
      if (ex_current_was_last) {
        list->last = new_element;
        ex_current_was_last = false;
      }
    }
    next = new_element;
  }
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_before_then_move
 *
 *  Add a new element to the list before the current element and move the
 *  iterator to the new element.
 **********************************************************************/

inline void ELIST_ITERATOR::add_before_then_move( // element to add
    ELIST_LINK *new_element) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_before_then_move", ABORT, nullptr);
  }
  if (!new_element) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_before_then_move", ABORT, "new_element is nullptr");
  }
  if (new_element->next) {
    STILL_LINKED.error("ELIST_ITERATOR::add_before_then_move", ABORT, nullptr);
  }
#endif

  if (list->empty()) {
    new_element->next = new_element;
    list->last = new_element;
    prev = next = new_element;
  } else {
    prev->next = new_element;
    if (current) { // not extracted
      new_element->next = current;
      next = current;
    } else { // current extracted
      new_element->next = next;
      if (ex_current_was_last) {
        list->last = new_element;
      }
      if (ex_current_was_cycle_pt) {
        cycle_pt = new_element;
      }
    }
  }
  current = new_element;
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_before_stay_put
 *
 *  Add a new element to the list before the current element but don't move the
 *  iterator to the new element.
 **********************************************************************/

inline void ELIST_ITERATOR::add_before_stay_put( // element to add
    ELIST_LINK *new_element) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_before_stay_put", ABORT, nullptr);
  }
  if (!new_element) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_before_stay_put", ABORT, "new_element is nullptr");
  }
  if (new_element->next) {
    STILL_LINKED.error("ELIST_ITERATOR::add_before_stay_put", ABORT, nullptr);
  }
#endif

  if (list->empty()) {
    new_element->next = new_element;
    list->last = new_element;
    prev = next = new_element;
    ex_current_was_last = true;
    current = nullptr;
  } else {
    prev->next = new_element;
    if (current) { // not extracted
      new_element->next = current;
      if (next == current) {
        next = new_element;
      }
    } else { // current extracted
      new_element->next = next;
      if (ex_current_was_last) {
        list->last = new_element;
      }
    }
    prev = new_element;
  }
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_list_after
 *
 *  Insert another list to this list after the current element but don't move
 *the
 *  iterator.
 **********************************************************************/

inline void ELIST_ITERATOR::add_list_after(ELIST *list_to_add) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_list_after", ABORT, nullptr);
  }
  if (!list_to_add) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_list_after", ABORT, "list_to_add is nullptr");
  }
#endif

  if (!list_to_add->empty()) {
    if (list->empty()) {
      list->last = list_to_add->last;
      prev = list->last;
      next = list->First();
      ex_current_was_last = true;
      current = nullptr;
    } else {
      if (current) { // not extracted
        current->next = list_to_add->First();
        if (current == list->last) {
          list->last = list_to_add->last;
        }
        list_to_add->last->next = next;
        next = current->next;
      } else { // current extracted
        prev->next = list_to_add->First();
        if (ex_current_was_last) {
          list->last = list_to_add->last;
          ex_current_was_last = false;
        }
        list_to_add->last->next = next;
        next = prev->next;
      }
    }
    list_to_add->last = nullptr;
  }
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_list_before
 *
 *  Insert another list to this list before the current element. Move the
 *  iterator to the start of the inserted elements
 *  iterator.
 **********************************************************************/

inline void ELIST_ITERATOR::add_list_before(ELIST *list_to_add) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_list_before", ABORT, nullptr);
  }
  if (!list_to_add) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_list_before", ABORT, "list_to_add is nullptr");
  }
#endif

  if (!list_to_add->empty()) {
    if (list->empty()) {
      list->last = list_to_add->last;
      prev = list->last;
      current = list->First();
      next = current->next;
      ex_current_was_last = false;
    } else {
      prev->next = list_to_add->First();
      if (current) { // not extracted
        list_to_add->last->next = current;
      } else { // current extracted
        list_to_add->last->next = next;
        if (ex_current_was_last) {
          list->last = list_to_add->last;
        }
        if (ex_current_was_cycle_pt) {
          cycle_pt = prev->next;
        }
      }
      current = prev->next;
      next = current->next;
    }
    list_to_add->last = nullptr;
  }
}

/***********************************************************************
 *                          ELIST_ITERATOR::extract
 *
 *  Do extraction by removing current from the list, returning it to the
 *  caller, but NOT updating the iterator.  (So that any calling loop can do
 *  this.)   The iterator's current points to nullptr.  If the extracted element
 *  is to be deleted, this is the callers responsibility.
 **********************************************************************/

inline ELIST_LINK *ELIST_ITERATOR::extract() {
  ELIST_LINK *extracted_link;

#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::extract", ABORT, nullptr);
  }
  if (!current) { // list empty or
                  // element extracted
    NULL_CURRENT.error("ELIST_ITERATOR::extract", ABORT, nullptr);
  }
#endif

  if (list->singleton()) {
    // Special case where we do need to change the iterator.
    prev = next = list->last = nullptr;
  } else {
    prev->next = next; // remove from list

    ex_current_was_last = (current == list->last);
    if (ex_current_was_last) {
      list->last = prev;
    }
  }
  // Always set ex_current_was_cycle_pt so an add/forward will work in a loop.
  ex_current_was_cycle_pt = (current == cycle_pt);
  extracted_link = current;
  extracted_link->next = nullptr; // for safety
  current = nullptr;
  return extracted_link;
}

/***********************************************************************
 *                          ELIST_ITERATOR::move_to_first()
 *
 *  Move current so that it is set to the start of the list.
 *  Return data just in case anyone wants it.
 **********************************************************************/

inline ELIST_LINK *ELIST_ITERATOR::move_to_first() {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::move_to_first", ABORT, nullptr);
  }
#endif

  current = list->First();
  prev = list->last;
  next = current ? current->next : nullptr;
  return current;
}

/***********************************************************************
 *                          ELIST_ITERATOR::mark_cycle_pt()
 *
 *  Remember the current location so that we can tell whether we've returned
 *  to this point later.
 *
 *  If the current point is deleted either now, or in the future, the cycle
 *  point will be set to the next item which is set to current.  This could be
 *  by a forward, add_after_then_move or add_after_then_move.
 **********************************************************************/

inline void ELIST_ITERATOR::mark_cycle_pt() {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::mark_cycle_pt", ABORT, nullptr);
  }
#endif

  if (current) {
    cycle_pt = current;
  } else {
    ex_current_was_cycle_pt = true;
  }
  started_cycling = false;
}

/***********************************************************************
 *                          ELIST_ITERATOR::at_first()
 *
 *  Are we at the start of the list?
 *
 **********************************************************************/

inline bool ELIST_ITERATOR::at_first() const {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::at_first", ABORT, nullptr);
  }
#endif

  // we're at a deleted
  return ((list->empty()) || (current == list->First()) ||
          ((current == nullptr) && (prev == list->last) && // NON-last pt between
           !ex_current_was_last));                         // first and last
}

/***********************************************************************
 *                          ELIST_ITERATOR::at_last()
 *
 *  Are we at the end of the list?
 *
 **********************************************************************/

inline bool ELIST_ITERATOR::at_last() const {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::at_last", ABORT, nullptr);
  }
#endif

  // we're at a deleted
  return ((list->empty()) || (current == list->last) ||
          ((current == nullptr) && (prev == list->last) && // last point between
           ex_current_was_last));                          // first and last
}

/***********************************************************************
 *                          ELIST_ITERATOR::cycled_list()
 *
 *  Have we returned to the cycle_pt since it was set?
 *
 **********************************************************************/

inline bool ELIST_ITERATOR::cycled_list() const {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::cycled_list", ABORT, nullptr);
  }
#endif

  return ((list->empty()) || ((current == cycle_pt) && started_cycling));
}

/***********************************************************************
 *                          ELIST_ITERATOR::sort()
 *
 *  Sort the elements of the list, then reposition at the start.
 *
 **********************************************************************/

inline void ELIST_ITERATOR::sort( // sort elements
    int comparator(               // comparison routine
        const void *, const void *)) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::sort", ABORT, nullptr);
  }
#endif

  list->sort(comparator);
  move_to_first();
}

/***********************************************************************
 *                          ELIST_ITERATOR::add_to_end
 *
 *  Add a new element to the end of the list without moving the iterator.
 *  This is provided because a single linked list cannot move to the last as
 *  the iterator couldn't set its prev pointer.  Adding to the end is
 *  essential for implementing
              queues.
**********************************************************************/

inline void ELIST_ITERATOR::add_to_end( // element to add
    ELIST_LINK *new_element) {
#ifndef NDEBUG
  if (!list) {
    NO_LIST.error("ELIST_ITERATOR::add_to_end", ABORT, nullptr);
  }
  if (!new_element) {
    BAD_PARAMETER.error("ELIST_ITERATOR::add_to_end", ABORT, "new_element is nullptr");
  }
  if (new_element->next) {
    STILL_LINKED.error("ELIST_ITERATOR::add_to_end", ABORT, nullptr);
  }
#endif

  if (this->at_last()) {
    this->add_after_stay_put(new_element);
  } else {
    if (this->at_first()) {
      this->add_before_stay_put(new_element);
      list->last = new_element;
    } else { // Iteratr is elsewhere
      new_element->next = list->last->next;
      list->last->next = new_element;
      list->last = new_element;
    }
  }
}

#define ELISTIZEH(CLASSNAME)                                                 \
  class CLASSNAME##_LIST : public X_LIST<ELIST, ELIST_ITERATOR, CLASSNAME> { \
  public:                                                                    \
    using X_LIST<ELIST, ELIST_ITERATOR, CLASSNAME>::X_LIST;                  \
  };                                                                         \
  class CLASSNAME##_IT : public X_ITER<ELIST_ITERATOR, CLASSNAME> {          \
  public:                                                                    \
    using X_ITER<ELIST_ITERATOR, CLASSNAME>::X_ITER;                         \
    CLASSNAME##_IT(CLASSNAME##_LIST *list) : X_ITER(list) {}                 \
  };

} // namespace tesseract

#endif
