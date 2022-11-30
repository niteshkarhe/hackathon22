import { ComponentFixture, TestBed } from '@angular/core/testing';

import { JsonTypeComponent } from './json-type.component';

describe('JsonTypeComponent', () => {
  let component: JsonTypeComponent;
  let fixture: ComponentFixture<JsonTypeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ JsonTypeComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(JsonTypeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
